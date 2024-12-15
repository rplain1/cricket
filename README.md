This is to analyze data from https://cricsheet.org/
Download the data in json format from here and here, into the `raw-data/` directory.

### Install dependencies
Use `uv` to setup the environment

```
pipx install uv
uv install
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
.
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
├── predictions.csv
├── pyproject.toml
├── raw-data
│   ├── innings_results.json
│   └── match_results.json
├── report
│   ├── ireland_first_5_overs.csv
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
