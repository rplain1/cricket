import os
import logging
import joblib
import polars as pl
import random
import extract
import transform
import load
import train_model
import predict

logging.basicConfig(level=logging.INFO)


def main():
    """
    Orchestrates the data pipeline: extraction, transformation, model training, and prediction.

    Steps:
        1. Extracts data
        2. Transforms the data
        3. Loads the data
        4. Selects and trains the model
        5. Runs predictions on new data

    Logs the progress and any errors during the pipeline execution.
    """
    random.seed(527)
    try:
        logging.info("Extracting data...")
        extract.main()

        logging.info("Transforming data...")
        transformed_data = transform.main()

        transformed_data = pl.read_parquet(os.path.join("data", "transformed", "transformed_data.parquet"))
        load.main(transformed_data)

        logging.info("Training model...")
        data = train_model.load_data()
        _model, _params = train_model.select_model(data)
        train_model.train_model(data, _model, _params)

        logging.info("Running predictions...")
        input_data = predict.load_data(os.path.join("report", "ireland_first_5_overs.csv"))
        model = joblib.load("models/model.pkl")
        output_data = predict.predict(model, input_data)

        print(output_data.head())

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
