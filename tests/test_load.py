import pytest
from load import time_series_split, split_feature_target
import polars as pl
import numpy as np


@pytest.mark.parametrize(
    "data",
    [
        pl.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-01", "2022-02-01", "2021-03-01"],
                "test_col": [x for x in range(4)],
                "test_col2": [x**x for x in range(4)],
            }
        ),
        pl.DataFrame(
            {
                "dates": np.random.randint(low=0, high=100, size=10_000),
                "test_col": [x for x in range(10_000)],
            }
        ),
        pl.DataFrame(
            {
                "dates": np.random.randint(low=0, high=100, size=527),
                "test_col": [x for x in range(527)],
            }
        ),
    ],
)
def test_time_series_split(data):
    split_data = time_series_split(data)
    assert isinstance(split_data, dict)
    assert all([x in split_data.keys() for x in ["train", "val", "test"]])

    nrow_train = data.filter(pl.col("dates").is_in(split_data["train"])).shape[0]
    assert nrow_train / data.shape[0] <= 0.7

    val_perc = data.filter(pl.col("dates").is_in(split_data["val"])).shape[0] / data.shape[0]
    test_perc = data.filter(pl.col("dates").is_in(split_data["test"])).shape[0] / data.shape[0]

    assert abs(val_perc - test_perc) < 0.04


@pytest.mark.parametrize(
    "data, should_raise",
    [
        (  # This is expected to pass
            pl.DataFrame(
                {
                    "runs": [x for x in range(4)],
                    "test_col": [x**x for x in range(4)],
                }
            ),
            False,
        ),
        (  # This is expected to raise an error (target column is missing)
            pl.DataFrame(
                {
                    "dates": np.random.randint(low=0, high=100, size=527),
                    "test_col": [x for x in range(527)],
                }
            ),
            True,
        ),
    ],
)
def test_split_feature_target(data, should_raise):
    if should_raise:
        with pytest.raises(AssertionError):
            X, y = split_feature_target(data)
    else:
        X, y = split_feature_target(data, target="runs")
        assert X.shape == (4, 1)
        assert y.shape == (4,)
