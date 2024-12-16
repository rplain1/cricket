This is to analyze data from https://cricsheet.org/

This takes the input data of matches results and innings results, and runs a pipeline performs:
- data ingestion
- transformation
- model preprocssesing
- model training
- runs model predictions on data sample

### Docker

To run the fully contained solution, you can execute via Docker.

#### Prerequisites
Ensure that Docker installed on your machine. If you don't have Docker, you can install it from [here](https://docs.docker.com/engine/install/).

#### Build the Docker Image
In your project directory, build the Docker image by running:

```
docker build -t cricket .
```

This command creates an image named cricket using the Dockerfile in the current directory.

#### Run the Docker Container
To run the container and execute the commands in the Dockerfile, use the following command:

```
docker run --it cricket
```
This is set to run everything needed to build the pipleine, and output sample model predictions.

### Run locally

If you want to bypass automatic download of the datasets, you can download the data in json format from [here](https://drive.google.com/file/d/19hVoi9f7n7etcmSXx7WHeiDp9pOLpQvN/view?usp=sharing) and [https://drive.google.com/file/d/1wQO9zr1VH8bY2W4Ca6cMxPdAoPOHo6X6/view?usp=sharing], into the `raw-data/` directory.

### Install dependencies
Use `uv` to setup the environment. You can find out how to install `uv` [here](https://docs.astral.sh/uv/getting-started/installation/#pypi).

```
uv python install
uv sync
uv venv source .venv/bin/activate
```

### Run the pipeine
```
uv run src/main.py
```

### Make Predictions
Once the pipeline has been run, you can make predictions with the following command:

```
python src/predict.py --data report/ireland_first_5_overs.csv --model models/model.pkl --output report/predictions.csv
```

### Run tests
```
uv run pytest
```

### Project Structure

Once the pipeline has been run, the following directories and sub-directories will populate to resemble the following:
```
.
├── Dockerfile
├── README.md
├── data
│   ├── extracted
│   │   ├── innings_results.parquet
│   │   ├── match_results.parquet
│   │   └── teams.csv
│   ├── model-prep
│   │   ├── test_features.pkl
│   │   ├── test_target.pkl
│   │   ├── train_features.pkl
│   │   ├── train_target.pkl
│   │   ├── val_features.pkl
│   │   └── val_target.pkl
│   └── transformed
│       └── transformed_data.parquet
├── models
│   └── model.pkl
├── pyproject.toml
├── raw-data
│   ├── innings_results.json
│   └── match_results.json
├── report
│   ├── ireland_first_5_overs.csv
│   ├── predictions.csv
│   └── q3a.csv
├── src
│   ├── __init__.py
│   ├── extract.py
│   ├── load.py
│   ├── main.py
│   ├── predict.py
│   ├── train_model.py
│   └── transform.py
├── tests
│   ├── __init__.py
│   ├── data
│   │   └── test_data.csv
│   ├── test_load.py
│   ├── test_predict.py
│   ├── test_train_model.py
│   └── test_transform.py
└── uv.lock
```
