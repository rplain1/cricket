import os
import logging
import joblib
import polars as pl
import extract
import transform
import load
import train_model
import predict

logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function to orchestrate the data pipeline: extract, transform, train, and predict.
    """
    try:
        # Step 1: Data Extraction
        logging.info("Extracting data...")
        extract.main()

        # Step 2: Data Transformation
        logging.info("Transforming data...")
        transformed_data = transform.main()

        # Step 3: Loading data
        transformed_data = pl.read_parquet(os.path.join("data", "transformed", "transformed_data.parquet"))
        load.main(transformed_data)

        # Step 3: Model Training
        logging.info("Training model...")
        data = train_model.load_data()
        _model, _params = train_model.select_model(data)
        train_model.train_model(data, _model, _params)

        # Step 5: Prediction
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
