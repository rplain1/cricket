import joblib
import polars as pl
import argparse


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_data(data_path):
    return pl.read_csv(data_path) if data_path.endswith(".csv") else pl.read_json(data_path)


def predict(model, data):
    features = data.drop("team")
    predictions = model.predict(features)

    return data.with_columns(pred_runs=predictions)


def main():
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
