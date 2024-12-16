import pytest
from load import split_data, split_feature_target
import polars as pl


@pytest.mark.parametrize(
    "data",
    [
        pl.DataFrame(
            {
                "test_col": [x for x in range(5)],
                "test_col2": [x**x for x in range(5)],
            }
        ),
        pl.DataFrame(
            {
                "test_col1": [x for x in range(10_000)],
                "test_col2": [x for x in range(10_000)],
            }
        ),
    ],
)
def test_split_data(data):
    """
    Tests the split_data function to ensure it returns a dictionary with 'train', 'val', and 'test' DataFrames.

    Args:
        data (pl.DataFrame): The input data to be split.

    Asserts:
        - Output is a dictionary
        - Dictionary contains 'train', 'val', and 'test' keys
        - Values for these keys are pl.DataFrame objects
    """
    data_dict = split_data(data)
    assert isinstance(data_dict, dict)
    assert all([x in data_dict.keys() for x in ["train", "val", "test"]])
    assert isinstance(data_dict["train"], pl.DataFrame)


@pytest.mark.parametrize(
    "data, should_raise",
    [
        (  # This is expected to pass
            pl.DataFrame(
                {
                    "test_col": [x for x in range(4)],
                    "runs": [x**x for x in range(4)],
                }
            ),
            False,
        ),
        (  # This is expected to raise an error (target column is missing)
            pl.DataFrame(
                {
                    "incorrect_col": [x for x in range(4)],
                    "this_isnt_runs": [x**x for x in range(4)],
                }
            ),
            True,
        ),
    ],
)
def test_split_feature_target(data, should_raise):
    """
    Tests the split_feature_target function to ensure proper feature and target splitting.

    Args:
        data (pl.DataFrame): The input data for splitting.
        should_raise (bool): Indicates if an error is expected.

    Asserts:
        - If should_raise is True, an AssertionError is raised.
        - Otherwise, verifies the shapes of the feature and target arrays.
    """
    if should_raise:
        with pytest.raises(AssertionError):
            X, y = split_feature_target(data)
    else:
        X, y = split_feature_target(data, target="test_col")
        assert X.shape == (4, 1)
        assert y.shape == (4,)
